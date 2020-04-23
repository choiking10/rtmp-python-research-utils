import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


BASE = "D:\\Users\\YunHoChoi\\project\\Node-Media-Server\\log\\remote\\log"
LOG_NAME = "2_25_16_34_to_earth_j10_1_delay"
FILE_FROM_EARTH = os.path.join(BASE, LOG_NAME, "black_from_earth.csv")
FILE_FROM_JUPITER = os.path.join(BASE, LOG_NAME, "black_from_jupiter.csv")


class GraphShower:
    COUNT = 100

    def __init__(self, xy1, xy2, start_at):
        fig = plt.figure()
        ax = fig.add_subplot()
        GraphShower.COUNT += 1

        t1 = np.array([value1 for value1, value2 in xy1[0]])
        s1 = np.array([value2 for value1, value2 in xy1[0]])

        t2 = np.array([value1 for value1, value2 in xy2[0]])
        s2 = np.array([value2 for value1, value2 in xy2[0]])
        ax.plot(t1, s1, label=xy1[1])
        ax.plot(t2, s2, label=xy2[1])
        ax.set_ylabel('total bytes(kb)')
        ax.set_xlabel('relative times(ms)')

        plt.title(LOG_NAME+"/"+start_at.strftime("%H:%M:%S.%f"))
        ax.legend()
        plt.show()


def parse_datetime(date, form="%d/%m/%Y %H:%M:%S"):
    result = datetime.strptime(' '.join(date.split(' ')[:-1]), form)
    result += timedelta(microseconds=int(date.split(' ')[-1])*1000)

    return result


class LogRow:
    LABEL = ["time", "tag", "chunk_id", "stream_id", "length", "clock", "bytes", "chunk_size", "total_bytes"]
    DATA_PARSER = [parse_datetime, str, int, int, int, int, int, int, int]

    def __init__(self, row):
        for i in range(0, len(LogRow.LABEL)):
            self.__setattr__(LogRow.LABEL[i], LogRow.DATA_PARSER[i](row[i]))

    def get_data_for_order(self, ordering):
        ret_data = []
        for order in ordering:
            ret_data.append(getattr(self, order))
        return tuple(ret_data), self

    def __gt__(self, other):
        return self.time < other.time and self.tag != other.tag

    def __str__(self):
        ret = ""
        for label in self.LABEL:
            if len(ret) != 0:
                ret += ", "
            ret += str(self.__getattribute__(label))
        return ret

    def __repr__(self):
        return self.__str__()


class Grouping:
    GROUP_BY = ["clock", "length", "chunk_id", "stream_id", "tag", ]
    ORDERING = ["time", "bytes"]
    FORMAT = "%H:%M:%S %f"

    def __init__(self, log_row):
        self.data = [log_row.get_data_for_order(Grouping.ORDERING)]
        self._from = log_row.time
        self._to = log_row.time

    def add_row(self, log_row):
        self.data.append(log_row.get_data_for_order(Grouping.ORDERING))
        self.data.sort()
        self._from = min(self._from, log_row.time)
        self._to = max(self._to, log_row.time)

    @property
    def length(self):
        return self._to - self._from

    @property
    def id(self):
        return Grouping.get_id(self.data[0][1])

    @property
    def tag(self):
        return self.data[0][1].tag

    @property
    def diff(self):
        diff = self._to - self._from
        return diff.seconds + diff.microseconds / 1000000

    @staticmethod
    def get_id(obj):
        return obj.get_data_for_order(Grouping.GROUP_BY)[0]

    def __str__(self):
        return str((self.tag, self._from.strftime(self.FORMAT), self._to.strftime(self.FORMAT)))

    def __repr__(self):
        return self.__str__()


class GroupingWithTag:
    GROUP_BY = ["clock", "length", "chunk_id", "stream_id", ]
    ORDERING = ["time", "bytes"]
    FORMAT = "%M:%S %f"

    def __init__(self, log_row):
        self.group_by_tag = {log_row.tag: Grouping(log_row)}
        self.id = GroupingWithTag.get_id(log_row)
        self.length = 1
        self.clock = log_row.clock
        self.length = log_row.length
        self.chunk_id = log_row.chunk_id
        self.stream_id = log_row.stream_id

    def add_row(self, log_row):
        if log_row.tag not in self.group_by_tag:
            self.group_by_tag[log_row.tag] = Grouping(log_row)
        else:
            self.group_by_tag[log_row.tag].add_row(log_row)
        self.length += 1

    def state(self):
        for i in self.group_by_tag.values():
            for j in self.group_by_tag.values():
                if i == j:
                    continue
                if (i._from <= j._from <= i._to or
                    i._from <= j._to <= i._to or
                    j._from <= i._from <= j._to or
                    j._from <= i._to <= j._to):
                    return "overlap"
        return "separate"

    def diff(self):
        for ikey, i in self.group_by_tag.items():
            for jkey, j in self.group_by_tag.items():
                if i == j:
                    continue
                if i._from < j._from:
                    delta = j._from - i._from
                else:
                    delta = i._from - j._from

                return (self.length, i.id[0], delta.seconds + delta.microseconds / 1000000, self.chunk_level_switch_cnt,
                        ikey, i.diff, i._from.strftime(self.FORMAT), i._to.strftime(self.FORMAT),
                        jkey, j.diff, j._from.strftime(self.FORMAT), j._to.strftime(self.FORMAT))
        return ()

    def chunk_level_diff(self):
        chunks = []
        for value in self.group_by_tag.values():
            chunks += value.data
        chunks.sort()

        return [(chunk.tag, chunk.length, chunk.chunk_size, chunk.time.strftime(self.FORMAT),) for key, chunk in chunks]

    def show_chunk_plt(self):
        chunks = []
        for value in self.group_by_tag.values():
            chunks += [(log_row.get_data_for_order(["time", "bytes"]), log_row)
                       for key, log_row in value.data]

        chunks.sort()

        earth = []
        jupiter = []
        start_row = chunks[0][1]
        start_time = start_row.time

        for key, log_row in chunks:
            if log_row.tag == 'black_from_earth':
                earth.append((
                    (log_row.time - start_time).total_seconds() * 1000, log_row.bytes / 1024))
            else:
                jupiter.append(((log_row.time - start_time).total_seconds() * 1000, log_row.bytes / 1024))
        additional_info = "{clock}, {length}".format(
            clock=start_row.clock,
            length=start_row.length
        )
        GraphShower(
            (earth, "earth ({}, {number_of})".format(additional_info, number_of=len(earth))),
            (jupiter, "jupiter ({}, {number_of})".format(additional_info, number_of=len(jupiter))),
            start_time
        )

    @property
    def chunk_level_switch_cnt(self):
        chunks = []
        for value in self.group_by_tag.values():
            chunks += [(log_row.get_data_for_order(["time", "bytes"]), log_row)
                       for key, log_row in value.data]

        chunks.sort()
        p_tag = chunks[0][1].tag
        switch_cnt = 0

        for key, log_row in chunks:
            if p_tag != log_row.tag:
                switch_cnt += 1
                p_tag = log_row.tag
        return switch_cnt

    @property
    def tags(self):
        return self.group_by_tag.values()

    @staticmethod
    def get_id(obj):
        return obj.get_data_for_order(GroupingWithTag.GROUP_BY)[0]

    def __str__(self):
        ret = "({}) ".format(self.length)
        for group in self.group_by_tag.values():
            if len(ret) != 0:
                ret += ", "
            ret += str(group)
        return ret

    def __repr__(self):
        return self.__str__()


class LogManager:
    def __init__(self, filenames):
        self.total_row = 0
        self.rows = []
        self.chunk_sizes = []
        self.message_sizes = []
        self.length_with_time = []
        self.diff_with_time = []
        self.jitter_with_time = []
        self.data_manager = {}

        bef = 0
        j_max, j_min, j_avg = 0, 10000000, 0
        e_max, e_min, e_avg = 0, 10000000, 0
        self.group_name = set()
        for filename in filenames:

            with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    log_row = LogRow(row)
                    if log_row.chunk_id != 5 or log_row.stream_id != 1:
                        continue
                    self.chunk_sizes.append(log_row.chunk_size)
                    self.rows.append(log_row)
                    self.group_name.update(log_row.tag)
                    if GroupingWithTag.get_id(log_row) not in self.data_manager:
                        self.data_manager[GroupingWithTag.get_id(log_row)] = GroupingWithTag(log_row)
                    else:
                        self.data_manager[GroupingWithTag.get_id(log_row)].add_row(log_row)
                    self.total_row += 1

        self.group_manager = [(key, value) for key, value in self.data_manager.items()]
        self.group_manager.sort()

        count = 0
        present_count = 0
        for key, val in self.group_manager:
            if val.id[0] == 0:
                continue
            if len(val.tags) == 1:
                value = list(val.group_by_tag.values())[0]
                print("only give me one!", val.length, value.id)
                continue

            if val.state() == "overlap":
                print("overlap!" + str(val.diff()))
                count += 1
                if val.chunk_level_switch_cnt >= 2:
                    pass
                    # val.show_chunk_plt()
            for group in val.group_by_tag.values():
                if group.data[0][1].tag == 'black_from_earth':
                    e_avg += group.diff
                    e_min = min(e_min, group.diff)
                    e_max = max(e_max, group.diff)

                else:
                    j_avg += group.diff
                    j_min = min(j_min, group.diff)
                    j_max = max(j_max, group.diff)

                    self.diff_with_time.append((val.clock, group.diff * 1000,))
                    self.length_with_time.append((val.clock, val.length / 1024,))
                    self.jitter_with_time.append((val.clock, group._to, ))

                    present_count += 1

                self.message_sizes.append(val.length)
                """
                for data in val.chunk_level_diff():
                    print(data)
                """

        print(self.total_row)
        print(count, len(self.group_manager))
        print("earth: {}/{}/{}".format(e_min, e_max, e_avg/len(self.group_manager)))
        print("jupiter: {}/{}/{}".format(j_min, j_max, j_avg/len(self.group_manager)))
        earth = []
        jupiter = []
        """
        for d in self.rows:
            if d.tag == 'black_from_earth':
                earth.append((d.clock, d.time))
            else:
                jupiter.append((d.clock, d.time))
        plt.hist(np.array(self.chunk_sizes), bins=20)
        plt.title("chunk_sizes")
        plt.xlabel("bytes")
        plt.ylabel("count")
        plt.show()

        plt.hist(np.array(self.message_sizes), bins=[pow(2, x) for x in range(0, 21)])
        plt.title("message_sizes")
        plt.xlabel("bytes")
        plt.ylabel("count")
        plt.show()
        """

        title = LOG_NAME
        show_range = slice(1000, 1500)

        fig = plt.figure()
        ax = fig.add_subplot()
        lengthw_with_time = self.length_with_time[show_range]

        t1 = np.array([value1 for value1, value2 in lengthw_with_time])
        s1 = np.array([value2 for value1, value2 in lengthw_with_time])
        ax.plot(t1, s1)
        #ax.set_ylabel('receive time (ms)')
        ax.set_ylabel('message size(length) (KB)')
        ax.set_xlabel('playback clock (ms)')
        ax.legend()
        plt.title(title)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot()

        diff_with_time = self.diff_with_time[show_range]
        t1 = np.array([value1 for value1, value2 in diff_with_time])
        s1 = np.array([value2 for value1, value2 in diff_with_time])
        ax.plot(t1, s1)
        ax.set_ylabel('receive complete time (ms)')
        ax.set_xlabel('playback clock (ms)')
        ax.legend()
        plt.title(title)
        plt.show()

        self.jitter_with_time.sort()
        jitter_with_time = []
        ac_jitter_with_time = []
        ac_needs_with_time = []
        bef = None
        nu = 0
        start_with = None
        for val1, val2 in self.jitter_with_time[show_range]:
            if not start_with:
                start_with = val1
            if bef:
                diff = val2 - bef[1]
                diff = diff.microseconds / 1000 + diff.seconds * 1000
                jitter_with_time.append((val1, diff))
                nu += diff
                ac_jitter_with_time.append((val1, nu))
                ac_needs_with_time.append((val1, val1 - start_with))

                if val2 < bef[1] or diff >= 1000000:
                    print(bef, val2)
            bef = val1, val2

        fig = plt.figure()
        ax = fig.add_subplot()

        t1 = np.array([value1 for value1, value2 in jitter_with_time])
        s1 = np.array([value2 for value1, value2 in jitter_with_time])
        ax.plot(t1, s1)
        ax.set_ylabel('jitter (ms)')
        ax.set_xlabel('playback clock (ms)')
        ax.legend()
        plt.title(title)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot()

        t1 = np.array([value1 for value1, value2 in ac_jitter_with_time])
        s1 = np.array([value2 for value1, value2 in ac_jitter_with_time])
        t2 = np.array([value1 for value1, value2 in ac_needs_with_time])
        s2 = np.array([value2 for value1, value2 in ac_needs_with_time])
        ax.plot(t1, s1, label="accumulate_jitter")
        ax.plot(t2, s2, label="accumulate_clock")
        ax.set_ylabel('accumulate time (ms)')
        ax.set_xlabel('playback clock (ms)')
        ax.legend()
        plt.title(title)
        plt.show()

        print("1")
        # GraphShower(earth, jupiter)



def main():

    LogManager([FILE_FROM_EARTH, FILE_FROM_JUPITER])


main()
